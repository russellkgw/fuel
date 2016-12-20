require_relative 'boot'

require 'rails/all'

# Require the gems listed in Gemfile, including any gems
# you've limited to :test, :development, or :production.
Bundler.require(*Rails.groups)

module Fuel
  class Application < Rails::Application
    # Settings in config/environments/* take precedence over those specified here.
    # Application configuration should go into files in config/initializers
    # -- all .rb files in that directory are automatically loaded.

    # Make use of structure.sql over schema.rb
    config.active_record.schema_format = :sql

    # Custom directories with classes and modules you want to be autoloadable.
    config.autoload_paths += [Rails.root.join('lib')]

    # Active job config
    # config.active_job.queue_adapter = :delayed_job

    config.action_mailer.delivery_method = :smtp
    config.action_mailer.smtp_settings = {
        :port           => 587,
        :address        => "smtp.mailgun.org",
        :domain         => 'sandbox811a6c5833b448dc8e930a427b6d557f.mailgun.org',
        :user_name      => 'postmaster@sandbox811a6c5833b448dc8e930a427b6d557f.mailgun.org',
        :password       => 'a995b798da44b6e7f0e9dc617c66c515',
        :authentication => :plain
    }

  end
end
