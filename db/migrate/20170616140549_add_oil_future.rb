class AddOilFuture < ActiveRecord::Migration[5.0]
  def change
    create_table(:oil_futures) do |t|
      t.string :currency, limit: 255
      t.date :date
      t.decimal :settle
      t.decimal :wave
      t.decimal :volume
      t.decimal :efp_volume
      t.decimal :efs_volume
      t.decimal :block_volume
      t.string :source, limit: 255
      t.timestamps
    end
  end
end
