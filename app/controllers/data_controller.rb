class DataController < ApplicationController
  before_action :authenticate

  def index
    @headers = { Data::Handler::EXCHANGE_RATE_FILE => Data::ExchangeRateImport::VALID_HEADERS,
                 Data::Handler::OIL_PRICE_FILE => Data::ExchangeRateImport::VALID_HEADERS,
                 Data::Handler::FUEL_PRICE_FILE => Data::ExchangeRateImport::VALID_HEADERS }
  end

  def create
    flash[:notice] = Data::Handler.process_csv_files(data_files)
    redirect_to action: 'index'
  end

  def show
    @data = build_data
  end

  private

  def data_files
    { Data::Handler::EXCHANGE_RATE_FILE => params[Data::Handler::EXCHANGE_RATE_FILE].try(:tempfile),
      Data::Handler::OIL_PRICE_FILE => params[Data::Handler::OIL_PRICE_FILE].try(:tempfile),
      Data::Handler::FUEL_PRICE_FILE => params[Data::Handler::FUEL_PRICE_FILE].try(:tempfile) }
  end

  def build_data
    { exchange_rates: ExchangeRate.order(date: :desc).first(10),
      oil_prices: OilPrice.order(date: :desc).first(10),
      fuel_prices: FuelPrice.order(date: :desc).first(10) }
  end

end